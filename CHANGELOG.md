# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Web reader for the paper at <https://thanatchon36.github.io/blp-research/paper/>, built by `make web`
  (`scripts/build_web.py`, pandoc + tectonic) from `paper/main.tex`.
- `pyproject.toml` (installable package in `src/blp/`), `Makefile`, CI on GitHub Actions, MIT license,
  Zenodo metadata and the README banner (`make banner`).
- `make arxiv` writes the self-contained arXiv source bundle `paper/arxiv.tar.gz` (`scripts/arxiv_bundle.py`).

### Changed
- The package moved from `blp/` to `src/blp/`; `requirements.txt` is replaced by `pyproject.toml`
  (`make setup`, optional extras `thai`, `llm`, `dev`).
- The paper compiles with tectonic (`make paper`).

## [0.1.0] - 2026-11-09

### Added
- Study 1: anonymization of interview notes, lexicon, LLM and ensemble coding against a gold standard
  (Cohen's kappa, Krippendorff's alpha, macro-F1), and sector-weighted, composition-adjusted diffusion indices.
- Study 2: expanding-window pseudo-real-time nowcasting with Clark–West and Diebold–Mariano tests.
- Study 3: retrieval-augmented drafting assistant with sentence-level citations and a verifier that blocks
  numbers it cannot trace to a data table and any identifying detail.
- Study 4: randomized crossover design, power simulation, analysis plan and time logging.
- Synthetic data generator, `run_pipeline.py`, `scripts/make_figures.py` and the preprint `paper/main.pdf`.
