#!/usr/bin/env python3
"""Export the Hugging Face dataset thanatchon36/blp-research into dataset/ (then `hf upload`).

Everything here is synthetic: the generator's notes, labels and target series (seed 42, as in the paper),
the simulated coders and experiment, and the pipeline's results on them. Notes are exported only after
masking (text_anon): the generator draws random 08x-xxx-xxxx mobile numbers that could belong to real people,
so the raw text stays out of the dataset; synthetic.generate(seed=42) reproduces it exactly.

Usage: make dataset   (writes dataset/; adds the Zenodo DOI once CITATION.cff has one)
       hf upload thanatchon36/blp-research dataset . --repo-type dataset
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

from blp import preprocess, synthetic  # noqa: E402
from blp.constants import REGION_EN  # noqa: E402
from blp.experiment import analysis, design  # noqa: E402

OUT = ROOT / "dataset"
SEED = 42
PHONE = r"(?<![\d.])0\d{2}-\d{3}-\d{4}(?!\d)|(?<![\d.])0[689]\d{8}(?!\d)"  # 08x-xxx-xxxx or 08xxxxxxxx


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)

    d = synthetic.generate(seed=SEED)
    anon = preprocess.Anonymizer(entities=d.entities)
    notes = d.notes.assign(text_anon=d.notes["raw_text"].map(anon.anonymize)).drop(columns="raw_text")
    leaks = int(notes["text_anon"].map(lambda t: len(anon.find_leaks(t))).sum())
    phones = int(notes["text_anon"].str.contains(PHONE).sum())
    if leaks or phones:
        raise SystemExit(f"masking left {leaks} names and {phones} phone numbers; nothing exported")
    notes.insert(3, "region_en", notes["region"].map(REGION_EN))
    notes.to_csv(OUT / "data" / "notes.csv", index=False)
    d.gold.to_csv(OUT / "data" / "gold_labels.csv", index=False)
    d.hard.assign(region_en=d.hard["region"].map(REGION_EN)).to_csv(OUT / "data" / "regional_activity.csv", index=False)
    pd.DataFrame(
        [{"region": r, "region_en": REGION_EN[r], "sector": s, "weight": w}
         for r, ws in d.sector_weights.items() for s, w in ws.items()]
    ).to_csv(OUT / "data" / "sector_weights.csv", index=False)
    synthetic.simulate_coders(d.gold, n_notes=400, seed=SEED).to_csv(OUT / "data" / "human_coders.csv", index=False)
    sched = design.crossover_schedule(design.make_participants(30, seed=SEED), seed=SEED)
    analysis.simulate_experiment(sched, true_reduction=0.30, seed=SEED).to_csv(
        OUT / "data" / "experiment_simulated.csv", index=False)

    # results: the pipeline's outputs plus the numbers and power grid quoted in the paper
    subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--seed", str(SEED), "--out", str(OUT / "results")],
                   check=True, capture_output=True)
    (OUT / "results" / "fig_indicators.png").unlink(missing_ok=True)
    for name in ("numbers.json", "power_grid.csv"):
        shutil.copyfile(ROOT / "paper" / name, OUT / "results" / name)
    for p in (OUT / "results").iterdir():
        text = p.read_text(encoding="utf-8")
        if pd.Series([text]).str.contains(PHONE).any():
            raise SystemExit(f"{p.name} contains a phone number; nothing exported")

    doi = re.search(r"^doi: (\S+)$", (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    card = (ROOT / "dataset_card" / "README.md").read_text(encoding="utf-8")
    if doi:
        card = card.replace("{DOI}", doi.group(1))
    else:  # no release archived on Zenodo yet: leave the DOI out
        card = re.sub(r";\narchived on Zenodo: \[\{DOI\}\]\(https://doi.org/\{DOI\}\)\.", ".", card)
        card = re.sub(r"\n  doi    = \{\{DOI\}\},", "", card)
    if "{DOI}" in card:
        raise SystemExit("dataset card still has a {DOI} placeholder")
    (OUT / "README.md").write_text(card, encoding="utf-8")
    shutil.copyfile(ROOT / "dataset_card" / "LICENSE", OUT / "LICENSE")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"README.md", "SHA256SUMS"})
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n" for p in files),
        encoding="utf-8")
    rows = {p.relative_to(OUT).as_posix(): len(pd.read_csv(p)) for p in files if p.suffix == ".csv"}
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
