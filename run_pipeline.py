#!/usr/bin/env python3
"""รัน pipeline ครบทุกบทด้วยข้อมูลจำลอง

    python run_pipeline.py                                  # ไม่ใช้ LLM (lexicon + extractive baseline)
    python run_pipeline.py --llm local --base-url http://gpu:8000/v1 --model <thai-llm>
    python run_pipeline.py --llm anthropic                  # ต้องตั้ง ANTHROPIC_API_KEY และได้รับอนุมัติข้อมูล

ผลลัพธ์ทั้งหมดอยู่ในโฟลเดอร์ outputs/ ตัวเลขจากข้อมูลจำลองใช้ตรวจว่าโค้ดทำงานถูก ไม่ใช่ผลวิจัย
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from blp import indicators, nowcast, preprocess, reliability, scoring, synthetic
from blp.assistant import drafter as drf
from blp.assistant import llm as llm_mod
from blp.assistant import retriever as retr
from blp.assistant import verifier as ver
from blp.constants import REGION_EN
from blp.experiment import analysis, design, power


def log(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="none", choices=["none", "local", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--llm-sample", type=int, default=100, help="จำนวนบันทึกที่ให้ LLM ลงรหัสเพื่อเทียบกับ gold")
    ap.add_argument("--region", default="เหนือ")
    ap.add_argument("--quarter", default="2025Q4")
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = llm_mod.make_llm(args.llm, model=args.model, base_url=args.base_url)
    summary: dict = {"llm": getattr(llm, "name", "none")}
    t0 = time.time()

    log("บทที่ 3: ข้อมูลจำลองและการปิดบังข้อมูล")
    data = synthetic.generate(seed=args.seed)
    anon = preprocess.Anonymizer(entities=data.entities)
    notes = data.notes.copy()
    notes["text_anon"] = notes["raw_text"].map(anon.anonymize)
    notes["text_body"] = notes["text_anon"].map(preprocess.body_text)
    leaks = int(notes["text_anon"].map(lambda t: len(anon.find_leaks(t))).sum())
    print(f"บันทึก {len(notes):,} ฉบับ | บริษัท {notes['firm_id'].nunique():,} แห่ง | "
          f"{notes['quarter'].min()}–{notes['quarter'].max()} | ข้อมูลระบุตัวตนที่เหลือหลังปิดบัง: {leaks}")
    print("ตัวอย่างก่อน/หลังปิดบัง:\n  " + notes["raw_text"].iloc[0].replace("\n", " / ")
          + "\n  " + notes["text_anon"].iloc[0].replace("\n", " / "))
    summary["n_notes"], summary["pii_left"] = len(notes), leaks

    log("บทที่ 4: ให้คะแนนบันทึกและความน่าเชื่อถือ")
    scored = scoring.score_notes(notes, scoring.LexiconScorer())
    rel = reliability.compare_to_gold(scored, data.gold)
    print("M1 lexicon เทียบ gold:\n" + rel.round(3).to_string(index=False))
    coders = synthetic.simulate_coders(data.gold, n_notes=400, seed=args.seed)
    k_h = reliability.cohen_kappa(coders["coder_a"], coders["coder_b"])
    a_h = reliability.krippendorff_alpha(np.vstack([coders["coder_a"], coders["coder_b"]]))
    print(f"คน-คน (จำลอง): Cohen's kappa = {k_h:.3f} | Krippendorff's alpha = {a_h:.3f}")
    rel.to_csv(out / "scoring_reliability.csv", index=False)
    summary["lexicon_kappa"] = float(rel.loc[rel["topic"] == "ALL", "kappa"].iloc[0])
    summary["lexicon_macro_f1"] = float(rel.loc[rel["topic"] == "ALL", "macro_f1"].iloc[0])
    summary["human_kappa"] = k_h
    if llm is not None:
        sample = notes.sample(min(args.llm_sample, len(notes)), random_state=args.seed)
        llm_scored = scoring.score_notes(sample, scoring.LLMScorer(llm))
        lex_sample = scored[scored["note_id"].isin(sample["note_id"])]
        comp = pd.DataFrame([{"method": "M1_lexicon", **reliability.compare_to_gold(lex_sample, data.gold).iloc[-1, 1:]},
                             {"method": "M3_llm", **reliability.compare_to_gold(llm_scored, data.gold).iloc[-1, 1:]}])
        print("เทียบบนตัวอย่างเดียวกัน:\n" + comp.round(3).to_string(index=False))
        comp.to_csv(out / "scoring_llm_vs_lexicon.csv", index=False)

    long = indicators.attach_metadata(scored, notes)
    ind = indicators.build_indicators(long, data.sector_weights)
    ind.to_csv(out / "indicators.csv", index=False)
    chk = ind.merge(data.hard, on=["region", "quarter"])
    corr = pd.DataFrame({v: chk.groupby("region").apply(lambda g, v=v: g[v].corr(g["activity"]), include_groups=False)
                         for v in ["BLP_RAI_raw", "BLP_RAI_w", "BLP_RAI"]})
    print("สหสัมพันธ์กับกิจกรรมเศรษฐกิจจริง (ดิบ / ถ่วงสาขา / ปรับ composition):\n" + corr.round(3).to_string())
    summary["corr_activity"] = corr.mean().round(3).to_dict()

    log("บทที่ 5: Nowcasting แบบ pseudo real-time")
    ds = nowcast.make_dataset(ind, data.hard)
    preds = nowcast.expanding_window_oos(ds, start="2018Q1", pooled=True)
    ev = nowcast.evaluate(preds)
    print(ev.round(3).to_string(index=False))
    print("lead-lag ของ OUTLOOK กับ y:\n" + nowcast.lead_lag_corr(ds, "OUTLOOK").round(3).to_string(index=False))
    preds.to_csv(out / "nowcast_predictions.csv", index=False)
    ev.to_csv(out / "nowcast_eval.csv", index=False)
    summary["nowcast"] = ev.set_index("model")[["rel_rmse"]].round(3)["rel_rmse"].to_dict()
    summary["nowcast_cw_p_M1"] = float(ev.loc[ev["model"] == "M1_AR+BLP", "cw_p"].iloc[0])

    log("บทที่ 6: AI assistant ร่างรายงานและตรวจร่าง")
    q = pd.Period(args.quarter, freq="Q")
    corpus = notes[notes["region"] == args.region]
    retriever = retr.HybridRetriever(corpus)
    drafter = drf.ReportDrafter(retriever, ind, llm=llm)
    report_md, drafts = drafter.draft_report(args.region, q)
    context_numbers = (q.quarter, q.year, q.year + 543)
    checks = {d.section: ver.verify_sentences(d.sentences, d.data_table, d.evidence, context_numbers, anon)
              for d in drafts}
    for section, rep in checks.items():
        print(f"{section:<18} passed={rep.passed} coverage={rep.coverage:.2f} "
              f"unverified={rep.unverified_numbers} invalid={rep.invalid_citations} pii={rep.pii_hits}")
    bad = drafts[0].sentences + ["ยอดขายของบริษัท ล้านนาฟู้ด1 จำกัด เพิ่มขึ้น 37.5 [D:D9]"]   # negative control
    neg = ver.verify_sentences(bad, drafts[0].data_table, drafts[0].evidence, context_numbers, anon)
    print(f"negative control (ใส่ตัวเลขปลอม+ชื่อบริษัท+อ้างอิงผิด): passed={neg.passed} -> "
          f"unverified={neg.unverified_numbers} invalid={neg.invalid_citations} pii={neg.pii_hits}")
    (out / "draft_report.md").write_text(report_md, encoding="utf-8")
    (out / "verification.json").write_text(json.dumps({s: r.to_dict() for s, r in checks.items()},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    summary["draft_all_passed"] = all(r.passed for r in checks.values())
    summary["negative_control_blocked"] = not neg.passed

    log("บทที่ 7: power analysis และซ้อมวิเคราะห์ผลการทดลอง")
    curve = power.power_curve(n_sims=args.n_sims)                                     # 4 งาน/คน
    curve8 = power.power_curve(n_tasks=8, n_sims=args.n_sims, reductions=(0.30,))     # ทางเลือกถ้าผลจริงแค่ 30%
    curves = pd.concat([curve, curve8], ignore_index=True)
    curves.to_csv(out / "power_curve.csv", index=False)
    print(curves.pivot_table(index="n_participants", columns=["n_tasks", "true_reduction"], values="power").round(2).to_string())
    for red in (0.30, 0.35):
        print(f"n ที่ให้ power >= 0.80 เมื่อผลจริงลดเวลา {red:.0%} (4 งาน/คน): {power.required_n(curve, red)}")
    print(f"n ที่ให้ power >= 0.80 เมื่อผลจริงลดเวลา 30% (8 งาน/คน): {power.required_n(curve8, 0.30)}")
    sched = design.crossover_schedule(design.make_participants(30, seed=args.seed), seed=args.seed)
    print("ความสมดุล งาน x เงื่อนไข:\n" + design.balance_table(sched).to_string())
    sim = analysis.simulate_experiment(sched, true_reduction=0.30, seed=args.seed)
    res = {"time": analysis.analyze_time(sim), "carryover": analysis.carryover_test(sim),
           "wilcoxon": analysis.participant_level_test(sim), "quality": analysis.analyze_quality(sim),
           "errors": analysis.analyze_errors(sim), "heterogeneity": analysis.heterogeneity(sim)}
    t = res["time"]
    print(f"dry run (ผลจริงตั้งไว้ 30%): ประมาณการลดเวลา {t['reduction']:.1%} "
          f"[95% CI {t['reduction_ci95'][0]:.1%}, {t['reduction_ci95'][1]:.1%}] "
          f"p(H0: ลด <= 20%) = {t['p_reduction_gt_threshold']:.4f}")
    print(f"คุณภาพ non-inferiority p = {res['quality']['p_noninferiority']:.4f} | "
          f"error IRR = {res['errors']['irr']:.2f} | carryover p = {res['carryover']['p']:.3f}")
    (out / "experiment_dryrun.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                                encoding="utf-8")
    summary["power_n_for_30pct"] = power.required_n(curve, 0.30)
    summary["power_n_for_35pct"] = power.required_n(curve, 0.35)
    summary["power_n_for_30pct_8tasks"] = power.required_n(curve8, 0.30)
    summary["dryrun_reduction"] = round(t["reduction"], 3)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for ax, (region, g) in zip(axes, chk.groupby("region")):
            x = g["quarter"].dt.to_timestamp()
            z = lambda s: (s - s.mean()) / s.std()
            ax.plot(x, z(g["activity"]), color="black", lw=2, label="True activity")
            ax.plot(x, z(g["BLP_RAI_raw"]), color="grey", ls="--", label="BLP raw")
            ax.plot(x, z(g["BLP_RAI"]), color="tab:blue", label="BLP adjusted")
            ax.set_title(f"{REGION_EN[region]} (synthetic, z-score)")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out / "fig_indicators.png", dpi=130)
    except ImportError:
        pass

    summary["runtime_sec"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nเสร็จใน {summary['runtime_sec']} วินาที ผลอยู่ใน {out.resolve()}")


if __name__ == "__main__":
    main()
