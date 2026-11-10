PY ?= .venv/bin/python
ARGS ?=

.PHONY: setup pipeline figures paper web arxiv banner test all clean

setup:          ## create venv and install the package (needs uv)
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev,thai]"

pipeline:       ## all four studies on synthetic data -> outputs/ (LLM options via ARGS="--llm local ...")
	$(PY) run_pipeline.py $(ARGS)

figures:        ## every data-driven figure and number in the paper -> paper/figures/, paper/numbers.json
	$(PY) scripts/make_figures.py

paper:          ## compile paper/main.pdf (tectonic) and rebuild the web reader docs/paper/
	cd paper && tectonic -X compile main.tex
	$(PY) scripts/build_web.py

web:            ## web reader only -> docs/paper/index.html (pandoc + tectonic; copies the current main.pdf)
	$(PY) scripts/build_web.py

arxiv: paper    ## arXiv source bundle -> paper/arxiv.tar.gz (prints title, abstract and comments for the form)
	$(PY) scripts/arxiv_bundle.py

banner:         ## README banner -> docs/assets/banner.png
	$(PY) scripts/make_banner.py

test:
	$(PY) -m pytest -q

all: pipeline figures paper

clean:
	rm -rf outputs paper/previews paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.blg paper/arxiv paper/arxiv.tar.gz
