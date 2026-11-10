# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- LinkedIn launch poster `docs/assets/linkedin_poster.png` (`make poster`, `scripts/make_linkedin_poster.py`) and the
  post in English and Thai, `docs/linkedin_post.md`.

## [0.1.1] - 2026-11-10

### Changed
- The paper no longer describes the program as doctoral research: the abstract and conclusion refer to the planned
  research, Figure 7 shows effort by component instead of by thesis chapter, and the timeline's milestones are the
  approval of the research proposal and the final report instead of the proposal and final defenses.

## [0.1.0] - 2026-11-10

### Added
- Study 1: anonymization of interview notes, lexicon, LLM and ensemble coding against a gold standard
  (Cohen's kappa, Krippendorff's alpha, macro-F1), and sector-weighted, composition-adjusted diffusion indices.
- Study 2: expanding-window pseudo-real-time nowcasting with Clark–West and Diebold–Mariano tests.
- Study 3: retrieval-augmented drafting assistant with sentence-level citations and a verifier that blocks
  numbers it cannot trace to a data table and any identifying detail.
- Study 4: randomized crossover design, power simulation, analysis plan and time logging.
- Synthetic data generator, `run_pipeline.py`, `scripts/make_figures.py` and the preprint `paper/main.pdf`.
- Web reader for the paper at <https://thanatchon36.github.io/blp-research/paper/>, built by `make web`
  (`scripts/build_web.py`, pandoc + tectonic) from `paper/main.tex`.
- `pyproject.toml` (installable package in `src/blp/`, extras `thai`, `llm`, `dev`), `Makefile`, CI on GitHub
  Actions, MIT license, Zenodo metadata and the README banner (`make banner`). The paper compiles with tectonic
  (`make paper`).
- `make arxiv` writes the self-contained arXiv source bundle `paper/arxiv.tar.gz` (`scripts/arxiv_bundle.py`).
- Dataset on Hugging Face, [`thanatchon36/blp-research`](https://huggingface.co/datasets/thanatchon36/blp-research):
  the synthetic notes (after masking), gold labels, target series, simulated coders and experiment, and the pipeline's results (`make dataset`).

