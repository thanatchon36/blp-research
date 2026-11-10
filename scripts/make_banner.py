#!/usr/bin/env python3
"""README banner -> docs/assets/banner.png.

Left: the composition-adjusted BLP regional activity index (BLP_RAI) for each region, built from the
synthetic notes by the same code as the paper. Right: name, one-line description, the four studies.
Synthetic data only: the lines show the software working, not the Thai regional economy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blp import indicators, preprocess, scoring, synthetic  # noqa: E402
from blp.constants import REGION_EN  # noqa: E402

BG, AMBER, PALE, MUTED, AXIS = "#03060d", "#f5a623", "#e8ecf4", "#9fb0d0", "#2b3c5c"

data = synthetic.generate(seed=42)
anon = preprocess.Anonymizer(entities=data.entities)
notes = data.notes.assign(text_anon=data.notes["raw_text"].map(anon.anonymize))
notes["text_body"] = notes["text_anon"].map(preprocess.body_text)
scored = scoring.score_notes(notes, scoring.LexiconScorer())
ind = indicators.build_indicators(indicators.attach_metadata(scored, notes), data.sector_weights)
regions = sorted(ind["region"].unique(), key=lambda r: REGION_EN.get(r, r))

fig = plt.figure(figsize=(16.8, 6.24), dpi=100, facecolor=BG)
n = len(regions)
for i, region in enumerate(regions):
    s = ind[ind["region"] == region].sort_values("quarter")
    ax = fig.add_axes([0.04, 0.09 + (n - 1 - i) * 0.84 / n, 0.27, 0.84 / n - 0.04], facecolor=BG)
    x = range(len(s))
    y = s["BLP_RAI"].to_numpy()
    ax.fill_between(x, y, y.min(), color=AMBER, alpha=0.12, lw=0)
    ax.plot(x, y, color=AMBER, lw=1.8)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.text(0, 1.0, REGION_EN.get(region, region), transform=ax.transAxes, color=MUTED, fontsize=10, va="top")
    ax.set_axis_off()

fig.text(0.37, 0.60, "blp-research", color=AMBER, fontsize=60, fontweight="bold", va="bottom")
fig.text(0.372, 0.38, "From liaison notes to regional indicators,\nand an AI assistant that drafts the report",
         color=PALE, fontsize=19, va="bottom", linespacing=1.6)
fig.text(0.372, 0.24, "text as data · diffusion indices · nowcasting · grounded drafting · randomized crossover",
         color=MUTED, fontsize=13.5, va="bottom")
fig.text(0.985, 0.035, "BLP regional activity index by region, 2012–2025 · synthetic data, not research findings",
         color="#7d8aa6", fontsize=9, ha="right")
out = ROOT / "docs" / "assets" / "banner.png"
fig.savefig(out, facecolor=BG)
print(out.relative_to(ROOT))
