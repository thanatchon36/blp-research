#!/usr/bin/env python3
"""LinkedIn poster -> docs/assets/linkedin_poster.png (portrait 4:5, 2160 x 2700).

The poster is laid out in HTML and photographed by headless Chrome; it is in English, with the example note and
the drafted sentences translated from the Thai the code produces.
Every number comes from paper/numbers.json; the note, its coding, the index and the drafted sentences are
produced by the project's own code on the synthetic data (seed 42). Requires Google Chrome.
"""
from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from blp import indicators, preprocess, scoring, synthetic  # noqa: E402

OUT = ROOT / "docs" / "assets" / "linkedin_poster.png"
W, H, SCALE = 1440, 1800, 1.5
CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]
TOPIC_EN = {"sales": "sales", "employment": "jobs", "investment": "investment", "price": "prices",
            "cost": "costs", "outlook": "outlook"}
ARROW = {1.0: "▲", 0.0: "▶", -1.0: "▼"}
DIR_CLASS = {1.0: "up", 0.0: "flat", -1.0: "down"}
# English renderings for the poster (the notes themselves are Thai); a missing entry stops the build
CLAUSE_EN = {
    "จำนวนพนักงานคงที่": "headcount unchanged",
    "ยอดขายโดยรวมไม่เปลี่ยนแปลงมากนัก": "overall sales little changed",
    "ต้นทุนการผลิตปรับลดลง": "production costs fell",
    "ลดงบลงทุนลงจากปีก่อน": "capital spending cut from last year",
    "มองแนวโน้มธุรกิจในระยะข้างหน้าเชิงลบ": "negative outlook for the business ahead",
}
SECTOR_EN = {"เกษตรแปรรูป": "agro-processing", "อุตสาหกรรมการผลิต": "manufacturing",
             "ท่องเที่ยวและโรงแรม": "tourism and hotels", "ค้าปลีกค้าส่ง": "retail and wholesale",
             "อสังหาริมทรัพย์และก่อสร้าง": "real estate and construction"}
PROVINCE_EN = {"เชียงใหม่": "Chiang Mai", "เชียงราย": "Chiang Rai", "ลำปาง": "Lampang", "พิษณุโลก": "Phitsanulok",
               "นครสวรรค์": "Nakhon Sawan"}

N = json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))
corr = N["corr_activity"]
corr_raw = np.mean([r["BLP_RAI_raw"] for r in corr.values()])
corr_adj = np.mean([r["BLP_RAI"] for r in corr.values()])
rmse_cut = 1 - N["nowcast"]["M1_AR+BLP"]["rel_rmse"]
power = N["power_grid"]["4tasks_35pct"]["30"]
ver = N["verifier"]

# --- the project's own code on the synthetic data -------------------------------------------------
d = synthetic.generate(seed=42)
anon = preprocess.Anonymizer(entities=d.entities)
notes = d.notes.assign(text_anon=d.notes["raw_text"].map(anon.anonymize))
notes["text_body"] = notes["text_anon"].map(preprocess.body_text)
lex = scoring.LexiconScorer()

# a recent North note with four or five coded topics, no distractor and every lexicon code right
gold = d.gold.pivot(index="note_id", columns="topic", values="label")
pick = None
for _, row in notes[(notes["region"] == "เหนือ") & (notes["quarter"].astype(str) == "2025Q4")].iterrows():
    clauses = preprocess.split_clauses(row["text_body"])
    topics = [scoring.detect_topic(c) for c in clauses]
    if None in topics or not 4 <= len(clauses) <= 5 or len(set(topics)) != len(topics):
        continue
    codes = lex.score_text(row["text_body"])
    g = gold.loc[row["note_id"]]
    if all((math.isnan(codes[t]) and math.isnan(g[t])) or codes[t] == g[t] for t in codes):
        pick = (row, clauses, topics, codes)
        break
if pick is None:
    raise SystemExit("no suitable example note")
row, clauses, topics, codes = pick
missing = [c for c in clauses if c not in CLAUSE_EN]
if missing:
    raise SystemExit(f"add English renderings for: {missing}")
header = (f"{row['interview_date']} · [FIRM] · {SECTOR_EN[row['sector']]} · {PROVINCE_EN[row['province']]} · "
          "contact: [PERSON], tel. [PHONE]")

scored = scoring.score_notes(notes, lex)
ind = indicators.build_indicators(indicators.attach_metadata(scored, notes), d.sector_weights)
north = ind[ind["region"] == "เหนือ"].sort_values("quarter").reset_index(drop=True)
act = d.hard[d.hard["region"] == "เหนือ"].sort_values("quarter")["activity"].to_numpy()


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


series = {"adj": z(north["BLP_RAI"].to_numpy()), "raw": z(north["BLP_RAI_raw"].to_numpy()), "act": z(act)}
corr_n_raw = np.corrcoef(series["raw"], series["act"])[0, 1]
corr_n_adj = np.corrcoef(series["adj"], series["act"])[0, 1]

# --- chart (inline SVG) ----------------------------------------------------------------------------
CW, CH, PL, PR, PT, PB = 1190, 330, 26, 8, 30, 32
lo = min(s.min() for s in series.values()) - 0.2
hi = max(s.max() for s in series.values()) + 0.2
n = len(act)


def px(i: int) -> float:
    return PL + i * (CW - PL - PR) / (n - 1)


def py(v: float) -> float:
    return PT + (hi - v) * (CH - PT - PB) / (hi - lo)


def path(s: np.ndarray) -> str:
    return "M" + " L".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(s))


quarters = north["quarter"].astype(str).tolist()
ticks = "".join(
    f'<text x="{px(i):.1f}" y="{CH - 8}" class="tick" text-anchor="middle">{q[:4]}</text>'
    for i, q in enumerate(quarters) if q.endswith("Q1") and int(q[:4]) % 2 == 0
)
covid = quarters.index("2020Q2")
area = path(series["adj"]) + f" L{px(n - 1):.1f},{py(lo):.1f} L{px(0):.1f},{py(lo):.1f} Z"
chart = f"""
<svg viewBox="0 0 {CW} {CH}" width="100%" role="img" aria-label="BLP activity index and latent activity, North region">
  <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
    <stop offset="0" stop-color="#f5a623" stop-opacity="0.28"/><stop offset="1" stop-color="#f5a623" stop-opacity="0"/>
  </linearGradient></defs>
  <line x1="{PL}" x2="{CW - PR}" y1="{py(0):.1f}" y2="{py(0):.1f}" class="zero"/>
  <path d="{area}" fill="url(#g)"/>
  <path d="{path(series['raw'])}" class="raw"/>
  <path d="{path(series['act'])}" class="act"/>
  <path d="{path(series['adj'])}" class="adj"/>
  <line x1="{px(covid):.1f}" x2="{px(covid):.1f}" y1="{PT - 4}" y2="{CH - PB + 4}" class="mark"/>
  <text x="{PL}" y="{PT - 10}" class="note">North region, standardized</text>
  <text x="{px(covid) + 10:.1f}" y="{PT - 10}" class="note">COVID-19 shock, 2020Q2</text>
  {ticks}
</svg>"""

# --- note card ---------------------------------------------------------------------------------------
spans = " ".join(f'<span class="hl hl-{t}">{html.escape(CLAUSE_EN[c])}</span>' for c, t in zip(clauses, topics))
chips = "".join(
    f'<span class="chip {DIR_CLASS[codes[t]]}"><b>{ARROW[codes[t]]}</b> {TOPIC_EN[t]}</span>'
    for t in topics
)
# a sentence the assistant drafted for the North, 2025Q4 (outputs/draft_report.md, make pipeline)
good_th = "ดัชนีการลงทุนปรับเพิ่มขึ้นจาก -35.9 ในไตรมาสก่อนเป็น 7.6"
good = "The investment index rose from −35.9 last quarter to 7.6"
draft = ROOT / "outputs" / "draft_report.md"
if draft.exists() and f"{good_th} [D:D1]" not in draft.read_text(encoding="utf-8"):
    raise SystemExit("outputs/draft_report.md changed; update the drafted sentence on the poster")

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
:root {{ --bg:#03060d; --panel:#0a1222; --rule:#1c2a44; --amber:#f5a623; --pale:#ffe8a3; --ink:#eef1f7;
  --ink2:#9fb0d0; --ink3:#6f7f9e; --up:#7dcfac; --down:#f08a6c; --flat:#9fb0d0; }}
* {{ box-sizing:border-box; margin:0; }}
html,body {{ width:{W}px; height:{H}px; background:var(--bg); color:var(--ink);
  font-family:"Avenir Next",sans-serif; -webkit-font-smoothing:antialiased; }}
body {{ background: radial-gradient(1100px 700px at 85% 18%, #0d1a33 0%, rgba(3,6,13,0) 70%), var(--bg); }}
.wrap {{ position:absolute; inset:0; padding:56px 96px 0; }}
.top {{ display:flex; gap:56px; align-items:baseline; font-size:26px; }}
.brand {{ color:var(--amber); font-weight:700; letter-spacing:.04em; }}
.tag {{ color:var(--ink2); font-weight:500; }}
.hero {{ display:grid; grid-template-columns: 560px 1fr; gap:40px; margin-top:26px; align-items:start; }}
.big {{ font-size:220px; line-height:.86; font-weight:700; color:var(--amber); letter-spacing:-.04em; }}
.big small {{ font-size:76px; color:var(--pale); font-weight:500; letter-spacing:0; margin-left:8px; }}
.lede {{ font-size:29px; line-height:1.36; margin-top:22px; font-weight:500; }}
.lede.amber {{ color:var(--amber); font-weight:600; margin-top:14px; }}
.card {{ background:var(--panel); border:1px solid var(--rule); border-radius:18px; padding:26px 28px; }}
.k {{ font-size:17px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink3); font-weight:600; }}
.k b {{ color:var(--amber); font-weight:700; margin-right:10px; }}
.meta {{ font-size:19px; color:var(--ink3); margin:14px 0 10px; }}
.body {{ font-size:27px; line-height:1.72; }}
.tr {{ font-size:16px; color:var(--ink3); font-style:italic; margin-top:6px; }}
.hl {{ border-radius:6px; padding:1px 6px; box-decoration-break:clone; -webkit-box-decoration-break:clone; }}
.hl-sales {{ background:rgba(245,166,35,.20); }} .hl-employment {{ background:rgba(125,207,172,.18); }}
.hl-investment {{ background:rgba(127,176,236,.20); }} .hl-price {{ background:rgba(240,138,108,.20); }}
.hl-cost {{ background:rgba(200,160,255,.18); }} .hl-outlook {{ background:rgba(255,232,163,.18); }}
.arrow {{ text-align:center; color:var(--ink3); font-size:22px; margin:14px 0 12px; letter-spacing:.1em; }}
.chips {{ display:flex; flex-wrap:wrap; gap:10px; }}
.chip {{ font-size:22px; font-weight:600; padding:7px 14px; border-radius:999px; border:1px solid var(--rule); background:#0d182c; }}
.chip.up b {{ color:var(--up); }} .chip.down b {{ color:var(--down); }} .chip.flat b {{ color:var(--flat); }}
.section {{ margin-top:30px; }}
.chart-head {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }}
.chart-head h3 {{ font-size:28px; font-weight:600; }}
.chart-head h3, .legend span {{ white-space:nowrap; }}
.legend .sub {{ color:var(--ink3); }}
.legend {{ display:flex; gap:20px; font-size:19px; color:var(--ink2); }}
.legend i {{ display:inline-block; width:28px; height:0; border-top:4px solid; vertical-align:middle; margin-right:8px; }}
svg .zero {{ stroke:#22314f; stroke-width:1.5; }} svg .mark {{ stroke:#3a4b6e; stroke-width:1.5; stroke-dasharray:4 5; }}
svg .adj {{ fill:none; stroke:var(--amber); stroke-width:4; stroke-linejoin:round; }}
svg .raw {{ fill:none; stroke:#5d6d8e; stroke-width:2.2; stroke-dasharray:7 6; }}
svg .act {{ fill:none; stroke:#e8ecf4; stroke-width:2.2; opacity:.85; }}
svg .tick {{ fill:var(--ink3); font-size:17px; font-family:"Avenir Next",sans-serif; }}
svg .note {{ fill:var(--ink2); font-size:17px; font-family:"Avenir Next",sans-serif; }}
.drafts {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.sent {{ font-size:22px; line-height:1.55; margin-top:10px; }}
.cite {{ font-family:"SF Mono",Menlo,monospace; font-size:18px; color:var(--amber); background:rgba(245,166,35,.12);
  padding:2px 7px; border-radius:6px; white-space:nowrap; }}
.bad .cite {{ color:var(--down); background:rgba(240,138,108,.14); }}
.bad .sent {{ color:#c3cbdc; }}
.verdict {{ margin-top:8px; font-size:19px; font-weight:600; }}
.ok {{ color:var(--up); }} .no {{ color:var(--down); }}
.stats {{ display:grid; grid-template-columns:1fr 1.25fr 1fr 1fr; gap:28px; margin-top:32px; border-top:1px solid var(--rule); padding-top:24px; }}
.stat b {{ display:block; font-size:60px; line-height:1; color:var(--amber); font-weight:700; letter-spacing:-.02em; }}
.stat span {{ display:block; font-size:20px; line-height:1.35; color:var(--ink2); margin-top:10px; }}
.stat i {{ font-style:normal; font-weight:500; color:var(--ink3); margin:0 4px; }}
.links {{ margin-top:26px; display:grid; grid-template-columns:1fr 1fr; gap:6px 40px; padding:20px 30px; }}
.links .k {{ font-size:15px; }} .links p {{ font-size:24px; margin-top:4px; }}
.links .fine {{ grid-column:1/-1; font-size:17px; color:var(--ink2); margin-top:8px; line-height:1.45; }}
.foot {{ position:absolute; left:96px; right:96px; bottom:28px; display:flex; justify-content:space-between; font-size:18px; color:var(--ink2); }}
</style></head><body><div class="wrap">
<div class="top"><span class="brand">BLP-RESEARCH</span><span class="tag">turning central bank interview notes into data</span></div>

<div class="hero">
  <div>
    <div class="big">4×<small>a year</small></div>
    <p class="lede">the Bank of Thailand hears from firms in every region. Official regional output arrives once a year, late.</p>
    <p class="lede amber">This research turns the interviews into indicators, and tests whether an AI assistant cuts the time to draft the regional report by more than 20%.</p>
  </div>
  <div class="card">
    <p class="k"><b>1</b>an interview note, masked</p>
    <p class="meta">{html.escape(header)}</p>
    <p class="body">{spans}</p>
    <p class="tr">translated from Thai</p>
    <p class="arrow">CODED BY THE LEXICON (M1) ▼</p>
    <div class="chips">{chips}</div>
  </div>
</div>

<div class="section card">
  <div class="chart-head"><h3><span class="k"><b>2</b></span>7,560 notes become a regional activity index</h3>
    <div class="legend"><span><i style="border-color:#f5a623"></i>adjusted index</span>
      <span><i style="border-color:#5d6d8e;border-top-style:dashed"></i>raw index</span>
      <span><i style="border-color:#e8ecf4"></i>true activity</span></div></div>
  {chart}
</div>

<div class="section drafts">
  <div class="card"><p class="k"><b>3</b>the assistant drafts, every sentence cited</p>
    <p class="sent">{html.escape(good)} <span class="cite">[D:D1]</span></p>
    <p class="verdict ok">✓ number found in data table D1</p></div>
  <div class="card bad"><p class="k"><b>✕</b>an injected error</p>
    <p class="sent">Sales at Lanna Food 1 Co., Ltd. rose 37.5 <span class="cite">[D:D9]</span></p>
    <p class="verdict no">✕ blocked: firm name, unverified number, invalid citation</p></div>
</div>

<div class="stats">
  <div class="stat"><b>{N['pii_left']}</b><span>firm or contact names left after masking {N['n_notes']:,} notes</span></div>
  <div class="stat"><b>{corr_raw:.2f}<i>→</i>{corr_adj:.2f}</b><span>correlation with true activity after composition adjustment</span></div>
  <div class="stat"><b>−{rmse_cut:.0%}</b><span>lower nowcast error than an autoregressive benchmark</span></div>
  <div class="stat"><b>{power:.2f}</b><span>power of a 30-economist crossover trial to detect a 35% time saving</span></div>
</div>

<div class="card links">
  <div><p class="k">Code + paper</p><p>github.com/thanatchon36/blp-research</p></div>
  <div><p class="k">Dataset</p><p>hf.co/datasets/thanatchon36/blp-research</p></div>
  <p class="fine">Every number here comes from synthetic data: it shows the software works, not that the hypotheses hold.
  The real interview notes are confidential and stay inside the Bank of Thailand.</p>
</div>
</div>
<div class="foot"><span>Thanatchon Chongmankhong · views are my own · DOI 10.5281/zenodo.24102338</span>
<span>synthetic data, seed 42</span></div>
</body></html>"""


def chrome() -> str:
    for c in CHROME:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Google Chrome not found")


with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "poster.html"
    src.write_text(page, encoding="utf-8")
    shot = Path(tmp) / "poster.png"
    subprocess.run(
        [chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
         f"--force-device-scale-factor={SCALE}", f"--screenshot={shot}", src.as_uri()],
        check=True, capture_output=True,
    )
    shutil.copyfile(shot, OUT)
    if "--html" in sys.argv:
        shutil.copyfile(src, OUT.with_suffix(".html"))
print(OUT.relative_to(ROOT), f"(North region: correlation {corr_n_raw:.2f} raw, {corr_n_adj:.2f} adjusted)")
