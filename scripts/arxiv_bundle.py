"""Build the arXiv source bundle, ``paper/arxiv.tar.gz``, from the paper as last built.

arXiv compiles the paper itself, from the root of the bundle, with pdfLaTeX (declared in
``00README.json``). This stages ``main.tex``, its bibliography as ``main.bbl`` and the figures it
uses, compiles the staged copy once with tectonic to prove it is self-contained, then tars it. It
also prints the title, the plain-text abstract (for the arXiv form) and a comments line. Run
``make paper`` first.

Usage: ``python scripts/arxiv_bundle.py``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE = PAPER / "arxiv"
OUT = PAPER / "arxiv.tar.gz"


def plain(latex: str) -> str:
    """LaTeX fragment -> one line of plain text (pandoc)."""
    out = subprocess.run(
        ["pandoc", "--from=latex", "--to=plain", "--wrap=none"], input=latex, capture_output=True, text=True, check=True
    ).stdout
    return " ".join(out.split())


def main() -> int:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "figures").mkdir(parents=True)
    shutil.copyfile(PAPER / "main.tex", STAGE / "main.tex")
    shutil.copyfile(PAPER / "main.bbl", STAGE / "main.bbl")
    figures = sorted(set(re.findall(r"\{figures/([\w.-]+)\}", tex)))
    for name in figures:
        shutil.copyfile(PAPER / "figures" / name, STAGE / "figures" / name)
    (STAGE / "00README.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "process": {"compiler": "pdflatex"},
                "sources": [{"filename": "main.tex", "usage": "toplevel"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # The staged copy must compile on its own. arXiv's pdfLaTeX reads main.bbl and never runs bibtex, but
    # tectonic always does, so the check compiles a copy that inputs main.bbl directly.
    check = STAGE.with_name("arxiv-check")
    if check.exists():
        shutil.rmtree(check)
    shutil.copytree(STAGE, check)
    checked = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", r"\\input{main.bbl}", tex)
    (check / "main.tex").write_text(checked, encoding="utf-8")
    subprocess.run(
        ["tectonic", "-X", "compile", "--keep-logs", "main.tex"], cwd=check, check=True, capture_output=True
    )
    log = (check / "main.log").read_text(encoding="utf-8", errors="replace")
    shutil.rmtree(check)
    pages = re.search(r"Output written on main\.(?:xdv|pdf) \((\d+) pages", log)
    if pages is None:
        raise SystemExit("staged compile left no page count")
    if re.search(r"Citation `[^']*' .*undefined|Reference `[^']*' .*undefined", log):
        raise SystemExit("staged compile has undefined citations or references")
    with tarfile.open(OUT, "w:gz") as tar:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(STAGE)))
    files = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file())
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} kB, {len(files)} files)")
    print("\n".join(f"  {f}" for f in files))

    # metadata for the submission form
    title = re.search(r"\{\\LARGE\\bfseries (.*?)\\par\}", tex, re.DOTALL).group(1).strip()
    subtitle = re.search(r"\{\\large\\bfseries (Text-Based.*?)\\par\}", tex, re.DOTALL).group(1)
    subtitle = re.sub(r"\\\\(\[[^]]*\])?\s*", " ", subtitle).strip()
    abstract = plain(re.search(r"\\begin\{quote\}\s*\\small\s*(.*?)\\end\{quote\}", tex, re.DOTALL).group(1))
    jel = re.search(r"\\textbf\{JEL classification\}\\quad (.*)", tex).group(1)
    jel = ", ".join(re.split(r"\s*\$\\cdot\$\s*", jel.strip()))
    n_fig = len(re.findall(r"\\begin\{figure\}", tex))
    n_tab = len(re.findall(r"\\begin\{table\}", tex))
    print(f"\nTitle:\n{title}: {subtitle}")
    print(f"\nAbstract ({len(abstract)} characters; arXiv allows 1,920):\n{abstract}")
    print(
        f"\nComments:\n{pages.group(1)} pages, {n_fig} figures, {n_tab} tables. Preprint; results are from "
        f"synthetic data. Code: https://github.com/thanatchon36/blp-research. JEL {jel}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
