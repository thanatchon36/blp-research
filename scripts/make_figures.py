#!/usr/bin/env python3
"""Reproduce every data-driven figure and number reported in the paper.

All numbers come from the synthetic data generator (blp/synthetic.py); they verify the
software, not the research hypotheses.  Output: paper/figures/*.pdf, paper/numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blp import indicators, nowcast, preprocess, reliability, scoring, synthetic  # noqa: E402
from blp.assistant import drafter as drf  # noqa: E402
from blp.assistant import retriever as retr  # noqa: E402
from blp.assistant import verifier as ver  # noqa: E402
from blp.constants import REGION_EN, REGIONS  # noqa: E402
from blp.experiment import analysis, design, power  # noqa: E402

FIG = ROOT / "paper" / "figures"
PREVIEW = ROOT / "paper" / "previews"
FIG.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)

INK, GREY, LIGHT = "#222222", "#8a8a8a", "#c8c8c8"
BLUE, BLUE_L, ORANGE = "#1f4e9c", "#8fb3e0", "#c45a00"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.edgecolor": "#444444", "axes.titlelocation": "left", "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.pad_inches": 0.03,
})
NUM: dict = {}


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=200)
    plt.close(fig)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


# ---------------------------------------------------------------- Study 1: data, coding, indices
data = synthetic.generate(seed=42)
anon = preprocess.Anonymizer(entities=data.entities)
notes = data.notes.copy()
notes["text_anon"] = notes["raw_text"].map(anon.anonymize)
notes["text_body"] = notes["text_anon"].map(preprocess.body_text)
NUM["n_notes"], NUM["n_firms"] = len(notes), int(notes["firm_id"].nunique())
NUM["pii_left"] = int(notes["text_anon"].map(lambda t: len(anon.find_leaks(t))).sum())

scored = scoring.score_notes(notes, scoring.LexiconScorer())
rel = reliability.compare_to_gold(scored, data.gold).set_index("topic")
NUM["lexicon"] = rel.round(3).to_dict(orient="index")
coders = synthetic.simulate_coders(data.gold, n_notes=400, seed=42)
NUM["human_kappa"] = round(reliability.cohen_kappa(coders["coder_a"], coders["coder_b"]), 3)
NUM["human_alpha"] = round(reliability.krippendorff_alpha(np.vstack([coders["coder_a"], coders["coder_b"]])), 3)

long = indicators.attach_metadata(scored, notes)
ind = indicators.build_indicators(long, data.sector_weights)
chk = ind.merge(data.hard, on=["region", "quarter"])
NUM["corr_activity"] = {REGION_EN[r]: {v: round(float(g[v].corr(g["activity"])), 3)
                                       for v in ("BLP_RAI_raw", "BLP_RAI_w", "BLP_RAI")}
                        for r, g in chk.groupby("region")}

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45), sharey=True, layout="constrained")
for ax, region in zip(axes, REGIONS):
    g = chk[chk["region"] == region].sort_values("quarter")
    x = g["quarter"].dt.to_timestamp()
    ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2020-07-01"), color="black", alpha=0.07, lw=0)
    ax.plot(x, zscore(g["activity"]), color=INK, lw=1.5)
    ax.plot(x, zscore(g["BLP_RAI_raw"]), color=GREY, lw=1.0, ls=(0, (4, 2)))
    ax.plot(x, zscore(g["BLP_RAI"]), color=BLUE, lw=1.2)
    ax.set_title(REGION_EN[region], pad=4)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(0, color=LIGHT, lw=0.6, zorder=0)
axes[0].set_ylabel("Standardized level (z-score)")
handles = [Line2D([], [], color=INK, lw=1.5), Line2D([], [], color=GREY, lw=1.0, ls=(0, (4, 2))),
           Line2D([], [], color=BLUE, lw=1.2)]
fig.legend(handles, ["Latent regional activity (truth)", "Raw diffusion index", "Composition-adjusted index"],
           loc="outside upper center", ncol=3, frameon=False)
save(fig, "fig_composition")

# ---------------------------------------------------------------- Study 2: nowcasting
ds = nowcast.make_dataset(ind, data.hard)
preds = nowcast.expanding_window_oos(ds, start="2018Q1", pooled=True)
ev = nowcast.evaluate(preds)
NUM["nowcast"] = ev.round(3).set_index("model").to_dict(orient="index")
NUM["nowcast_n_forecasts"] = int(len(preds))
NUM["leadlag_outlook"] = nowcast.lead_lag_corr(ds, "OUTLOOK").round(3).set_index("lead_k")["corr"].to_dict()

fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.55), layout="constrained", gridspec_kw={"width_ratios": [2.15, 1]})
p = preds[preds["region"] == "เหนือ"].sort_values("quarter")
x = p["quarter"].dt.to_timestamp()
a.plot(x, p["y"], color=INK, lw=1.5, marker="o", ms=2.2, label="Outturn")
a.plot(x, p["M0_AR"], color=GREY, lw=1.0, ls=(0, (4, 2)), label="M0: AR(1)")
a.plot(x, p["M2_AR+BLP+OUTLOOK"], color=BLUE, lw=1.2, label="M2: AR + BLP-RAI + outlook")
a.axhline(0, color=LIGHT, lw=0.6, zorder=0)
a.set_title("(a) Nowcasts for the North region", pad=4)
a.set_ylabel("Growth, % y/y (synthetic)")
a.xaxis.set_major_locator(mdates.YearLocator(2))
a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, handlelength=2.2, columnspacing=1.2)
labels = {"M0_AR": "M0: AR(1)", "M1_AR+BLP": "M1: + BLP-RAI", "M2_AR+BLP+OUTLOOK": "M2: + outlook"}
order = ["M2_AR+BLP+OUTLOOK", "M1_AR+BLP", "M0_AR"]
vals = ev.set_index("model").loc[order, "rel_rmse"]
b.barh([labels[m] for m in order], vals, color=[BLUE, BLUE_L, LIGHT], height=0.55)
for yy, v in enumerate(vals):
    b.text(v + 0.03, yy, f"{v:.2f}", va="center", ha="left", fontsize=8)
b.axvline(0.90, color=INK, lw=0.7, ls=":")
b.set_xlim(0, 1.25)
b.set_xticks([0, 0.5, 1.0])
b.set_xlabel("RMSE relative to M0")
b.set_title("(b) Relative RMSE, all regions", pad=4)
b.spines["left"].set_visible(False)
b.tick_params(axis="y", length=0)
save(fig, "fig_nowcast")

# ---------------------------------------------------------------- Study 3: assistant verification
q = pd.Period("2025Q4", freq="Q")
drafter = drf.ReportDrafter(retr.HybridRetriever(notes[notes["region"] == "เหนือ"]), ind, llm=None)
_, drafts = drafter.draft_report("เหนือ", q)
ctx = (q.quarter, q.year, q.year + 543)
reps = [ver.verify_sentences(d.sentences, d.data_table, d.evidence, ctx, anon) for d in drafts]
neg = ver.verify_sentences(drafts[0].sentences + ["Sales of บริษัท ล้านนาฟู้ด1 จำกัด rose 37.5 [D:D9]"],
                           drafts[0].data_table, drafts[0].evidence, ctx, anon)
NUM["verifier"] = {"sections": len(reps), "passed": sum(r.passed for r in reps),
                   "sentences": sum(r.n_sentences for r in reps), "coverage_min": min(r.coverage for r in reps),
                   "negative_control_passed": neg.passed, "neg_unverified": neg.unverified_numbers,
                   "neg_invalid": neg.invalid_citations, "neg_pii": neg.pii_hits}

# ---------------------------------------------------------------- Study 4: power and dry run
grid = pd.concat([power.power_curve(ns=(10, 15, 20, 25, 30, 35, 40), reductions=(0.25, 0.30, 0.35, 0.40),
                                    n_tasks=k, n_sims=4000, seed=11) for k in (4, 6, 8)], ignore_index=True)
grid.to_csv(ROOT / "paper" / "power_grid.csv", index=False)
NUM["power_grid"] = {f"{k}tasks_{int(r * 100)}pct": g.set_index("n_participants")["power"].round(2).to_dict()
                     for (k, r), g in grid.groupby(["n_tasks", "true_reduction"])}

fig, ax = plt.subplots(figsize=(6.5, 2.75), layout="constrained")
series = [(4, 0.25, BLUE_L, "-", "25%, 4 tasks"), (4, 0.30, "#4f7fc4", "-", "30%, 4 tasks"),
          (4, 0.35, BLUE, "-", "35%, 4 tasks"), (8, 0.30, ORANGE, (0, (4, 2)), "30%, 8 tasks")]
for k, r, c, ls, lab in series:
    g = grid[(grid["n_tasks"] == k) & (grid["true_reduction"] == r)]
    ax.plot(g["n_participants"], g["power"], color=c, ls=ls, lw=1.4, marker="o", ms=3, label=lab)
ax.axhline(0.80, color=GREY, lw=0.7, ls=":")
ax.axvline(30, color=GREY, lw=0.7, ls=":")
ax.set_xlim(9, 41)
ax.set_ylim(0, 1.02)
ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
ax.set_xlabel("Participants (n)")
ax.set_ylabel("Power, one-sided test vs. 20%")
ax.legend(title="True time reduction, tasks per person", loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=False, title_fontsize=8, alignment="left")
save(fig, "fig_power")


def quality_power(n=30, n_tasks=4, sd=1.0, sd_task=0.3, margin=0.5, true_diff=0.0, sims=4000, seed=3):
    rng = np.random.default_rng(seed)
    half = n_tasks // 2
    ai = rng.normal(true_diff, sd, (sims, n, half)).mean(axis=2) + rng.normal(0, sd_task, (sims, n, half)).mean(axis=2)
    ctrl = rng.normal(0, sd, (sims, n, half)).mean(axis=2) + rng.normal(0, sd_task, (sims, n, half)).mean(axis=2)
    d = ai - ctrl
    t = (d.mean(axis=1) + margin) / (d.std(axis=1, ddof=1) / np.sqrt(n))
    from scipy import stats
    return float((stats.t.sf(t, n - 1) < 0.05).mean())


NUM["quality_power_n30"] = round(quality_power(), 3)

sched = design.crossover_schedule(design.make_participants(30, seed=42), seed=42)
sim = analysis.simulate_experiment(sched, true_reduction=0.30, seed=42)
t = analysis.analyze_time(sim)
NUM["dryrun"] = {"reduction": round(t["reduction"], 3), "ci_lo": round(t["reduction_ci95"][0], 3),
                 "ci_hi": round(t["reduction_ci95"][1], 3), "lower_one_sided": round(t["reduction_lower_bound_one_sided"], 3),
                 "p_gt20": round(t["p_reduction_gt_threshold"], 4),
                 "quality_p": round(analysis.analyze_quality(sim)["p_noninferiority"], 3),
                 "quality_diff": round(analysis.analyze_quality(sim)["diff"], 3),
                 "irr": round(analysis.analyze_errors(sim)["irr"], 3), "carryover_p": round(analysis.carryover_test(sim)["p"], 3),
                 "wilcoxon_p": round(analysis.participant_level_test(sim)["p_wilcoxon"], 4),
                 "balance": design.balance_table(sched).to_dict()}

# ---------------------------------------------------------------- Work plan: effort and timeline
chapters = [("Ch. 1  Introduction", 5, 0), ("Ch. 2  Literature and framework", 10, 0),
            ("Ch. 3  Design, data and governance", 7, 3), ("Ch. 4  Study 1: indicator construction", 12, 13),
            ("Ch. 5  Study 2: validation and nowcasting", 8, 7), ("Ch. 6  Study 3: AI assistant", 6, 14),
            ("Ch. 7  Study 4: time experiment", 7, 3), ("Ch. 8  Discussion and conclusion", 5, 0)]
fig, ax = plt.subplots(figsize=(6.5, 2.85), layout="constrained")
for i, (name, acad, code) in enumerate(chapters):
    ax.barh(i, acad, color=LIGHT, height=0.62)
    ax.barh(i, code, left=acad, color=BLUE, height=0.62)
    ax.text(acad / 2, i, f"{acad}", ha="center", va="center", fontsize=7.5, color=INK)
    if code:
        ax.text(acad + code / 2, i, f"{code}", ha="center", va="center", fontsize=7.5, color="white")
    ax.text(acad + code + 0.5, i, f"{acad + code}%", ha="left", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(chapters)), [c[0] for c in chapters])
ax.invert_yaxis()
ax.set_xlim(0, 28.5)
ax.set_xlabel("Share of total thesis effort (%)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend([Patch(color=LIGHT), Patch(color=BLUE)], ["Academic design (60%)", "Code (40%)"],
          loc="lower right", frameon=False)
save(fig, "fig_effort")

rows = [("Literature review and proposal", 0, 6, "x"), ("Data access, ethics and governance", 3, 9, "x"),
        ("Gold-standard annotation", 6, 12, "a"), ("Study 1: indicator construction", 9, 18, "a"),
        ("Study 2: nowcasting evaluation", 15, 24, "a"), ("Study 3: assistant, three DSR cycles", 18, 30, "b"),
        ("Study 4: crossover experiment", 30, 36, "b"), ("Field pilot, two report cycles", 36, 42, "b"),
        ("Thesis writing and papers", 33, 46, "x")]
milestones = [("M1", 6), ("M2", 9), ("M3", 30), ("M4", 47)]
colors = {"x": LIGHT, "a": BLUE_L, "b": BLUE}
fig, ax = plt.subplots(figsize=(6.5, 3.05), layout="constrained")
for yr in range(4):
    if yr % 2 == 0:
        ax.axvspan(12 * yr, 12 * (yr + 1), color="black", alpha=0.035, lw=0, zorder=0)
for i, (name, s, e, kind) in enumerate(rows, start=1):
    ax.barh(i, e - s, left=s, color=colors[kind], height=0.56, zorder=2)
ax.plot([m for _, m in milestones], [0] * len(milestones), "D", color=ORANGE, ms=5.5, zorder=3)
for lab, m in milestones:
    ax.text(m, -0.62, lab, ha="center", va="bottom", fontsize=7.5, color=INK)
ax.set_yticks(range(0, len(rows) + 1), ["Milestones"] + [r[0] for r in rows])
ax.set_ylim(len(rows) + 0.7, -1.25)
ax.set_xlim(0, 48)
ax.set_xticks([6, 18, 30, 42], ["Year 1", "Year 2", "Year 3", "Year 4"])
ax.set_xticks(range(0, 49, 3), minor=True)
ax.tick_params(axis="x", which="major", length=0, pad=4)
ax.tick_params(axis="x", which="minor", length=2)
ax.tick_params(axis="y", length=0)
for xx in (12, 24, 36):
    ax.axvline(xx, color=GREY, lw=0.5, zorder=1)
ax.spines["left"].set_visible(False)
fig.legend([Patch(color=BLUE_L), Patch(color=BLUE), Patch(color=LIGHT)],
           ["Measurement track", "Report-production track", "Cross-cutting"],
           loc="outside lower center", ncol=3, frameon=False)
save(fig, "fig_timeline")

(ROOT / "paper" / "numbers.json").write_text(json.dumps(NUM, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(NUM, indent=1, ensure_ascii=False, default=str))
