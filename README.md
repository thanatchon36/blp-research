# From Liaison Notes to Regional Indicators

**Thanatchon Chongmankhong** · Preprint, November 2026 · [PDF](paper/main.pdf)

*Text-Based Measurement of Regional Economic Activity from Business Liaison Program Data and a Randomized
Evaluation of a Generative-AI Report-Drafting Assistant*

This repository contains the paper and the code for a four-study research program on the Bank of Thailand's
Business Liaison Program (BLP):

1. **Study 1** converts Thai-language interview notes into regional indicators (lexicon, fine-tuned and LLM coding;
   sector-weighted and composition-adjusted diffusion indices).
2. **Study 2** tests their predictive validity in pseudo-real-time nowcasting (Clark–West, Diebold–Mariano).
3. **Study 3** builds a retrieval-augmented drafting assistant that cites a source for every sentence and blocks
   numbers it cannot verify.
4. **Study 4** is a randomized crossover experiment testing whether the assistant cuts report-preparation time by
   more than 20% without lowering quality.

> **Synthetic data only.** BLP data are confidential and are not part of this repository. Everything here runs on a
> synthetic data generator; the numbers it produces verify the software and are not research findings.

## Repository structure

```
blp-research/
├── blp/                    # Python package
│   ├── synthetic.py        # synthetic BLP notes, gold labels, target series
│   ├── preprocess.py       # anonymization, Thai clause splitting
│   ├── scoring.py          # lexicon (M1), LLM (M3), ensemble (M4) scorers
│   ├── reliability.py      # Cohen's kappa, Krippendorff's alpha, macro-F1
│   ├── indicators.py       # diffusion, sector-weighted, composition-adjusted indices
│   ├── nowcast.py          # expanding-window evaluation, Clark–West, Diebold–Mariano
│   ├── assistant/          # LLM clients, hybrid retrieval, drafter, verifier
│   └── experiment/         # crossover design, power simulation, analysis, time logs
├── scripts/make_figures.py # regenerates every figure and number in the paper
├── run_pipeline.py         # runs all four studies end to end
├── tests/                  # smoke tests
└── paper/                  # LaTeX source, bibliography, figures and compiled PDF
```

| Module | Paper section |
|---|---|
| `blp/preprocess.py` | 3.4 Data governance and ethics |
| `blp/scoring.py`, `blp/reliability.py` | 4.2 Scoring methods, 4.5 Validity |
| `blp/indicators.py` | 4.3 Index construction, 4.4 Composition adjustment |
| `blp/nowcast.py` | 5 Study 2 |
| `blp/assistant/` | 6 Study 3 |
| `blp/experiment/` | 7 Study 4 |
| `blp/synthetic.py` | 8 Proof of concept, Appendix D |

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py              # all studies on synthetic data, results in outputs/
python scripts/make_figures.py      # paper figures -> paper/figures, numbers -> paper/numbers.json
python -m pytest -q                 # smoke tests
```

Build the paper (pdfLaTeX):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Using an LLM

The default deployment is an on-premise open-weight model behind an OpenAI-compatible endpoint (vLLM, Ollama):

```bash
python run_pipeline.py --llm local --base-url http://<gpu-host>:8000/v1 --model <thai-llm>
```

The Claude API client (`--llm anthropic`, reads `ANTHROPIC_API_KEY`) may only be used with anonymized data under an
approved data-processing agreement. Freeze model versions for the whole experiment.

## Plugging in real data (inside the secure enclave only)

* `notes`: `note_id, firm_id, region, province, sector, quarter (pandas Period), interview_date, raw_text`
* `hard`: `region, quarter, y` (official target series)
* `entities`: firm and contact names from the BLP contact database, used for masking and by the verifier
* sector weights: GRP sector shares by region (`blp/constants.py` holds illustrative values only)

Never commit confidential data: `data/` and `.env` are excluded by `.gitignore`.

## Citation

```bibtex
@misc{chongmankhong2026liaison,
  author = {Chongmankhong, Thanatchon},
  title  = {From Liaison Notes to Regional Indicators: Text-Based Measurement of Regional Economic Activity
            from Business Liaison Program Data and a Randomized Evaluation of a Generative-AI
            Report-Drafting Assistant},
  year   = {2026},
  note   = {Preprint}
}
```
