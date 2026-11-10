"""Smoke tests: python -m pytest -q"""
from blp import indicators, preprocess, reliability, scoring, synthetic
from blp.assistant import verifier as ver
from blp.experiment import design


def test_pipeline_small():
    d = synthetic.generate(seed=1, start="2020Q1", end="2022Q4", firms_per_quarter=10)
    anon = preprocess.Anonymizer(entities=d.entities)
    notes = d.notes.assign(text_anon=d.notes["raw_text"].map(anon.anonymize))
    assert notes["text_anon"].map(lambda t: len(anon.find_leaks(t))).sum() == 0
    notes["text_body"] = notes["text_anon"].map(preprocess.body_text)
    sc = scoring.score_notes(notes, scoring.LexiconScorer())
    rel = reliability.compare_to_gold(sc, d.gold)
    assert rel["kappa"].between(-1, 1).all()
    ind = indicators.build_indicators(indicators.attach_metadata(sc, notes), d.sector_weights)
    assert {"BLP_RAI", "OUTLOOK", "UNCERTAINTY"} <= set(ind.columns)


def test_verifier_blocks_bad_draft():
    table = {"D1": {"label": "x", "value": 1.5, "prev": 1.0, "change": 0.5}}
    ev = [{"ids": ["N1"], "dates": ["2025-01-01"], "sector": "s", "snippet": "ยอดขายเพิ่มขึ้น"}]
    assert ver.verify_sentences(["ดัชนีอยู่ที่ 1.5 [D:D1]"], table, ev).passed
    assert not ver.verify_sentences(["ดัชนีอยู่ที่ 9.9 [D:D2]"], table, ev).passed


def test_crossover_balanced():
    s = design.crossover_schedule(design.make_participants(16, seed=0), seed=0)
    tab = design.balance_table(s)
    assert (tab["AI"] == tab["CTRL"]).all()
