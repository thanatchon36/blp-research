#!/usr/bin/env python3
"""Build the web reader docs/paper/ from paper/main.tex (same text and numbers as the PDF).

tectonic compiles the paper once in a temporary directory to obtain the section, table and figure
numbers (main.aux); each TikZ diagram is compiled on its own and, like the matplotlib figures, converted
to SVG; pandoc renders the body with paper/reader.html (masthead, abstract, a results strip whose
numbers come from paper/numbers.json, contents, numbered sections, references and appendices). The
committed paper/main.pdf and the figures are copied next to the page; GitHub Pages serves docs/ at
https://thanatchon36.github.io/blp-research/. Requires pandoc, tectonic and pdftocairo (poppler) on PATH.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "docs" / "paper"
REPO_URL = "https://github.com/thanatchon36/blp-research"
PAPER_URL = "https://thanatchon36.github.io/blp-research/paper/"
DATA_URL = "https://huggingface.co/datasets/thanatchon36/blp-research"

# (label, style, finding, detail); {key.path} is looked up in paper/numbers.json.
# style: "ref" = result on the synthetic data, "null" = pending.
RESULTS = [
    ("Study 1 · coding", "ref",
     "Lexicon coder κ = {lexicon.ALL.kappa} against the gold standard",
     "macro-F1 {lexicon.ALL.macro_f1} over six topics; two simulated human coders agree at κ = {human_kappa}"),
    ("Study 1 · index", "ref",
     "Composition adjustment raises the correlation with latent activity from {corr_raw} to {corr_adj}",
     "mean over the Northeast, North and South; {n_notes} notes from {n_firms} firms, {pii_left} identifiers left after masking"),
    ("Study 2 · nowcast", "ref",
     "AR + BLP index cuts RMSE to {nowcast.M1_AR+BLP.rel_rmse} of the AR benchmark",
     "{nowcast.M2_AR+BLP+OUTLOOK.rel_rmse} with the outlook index; Clark–West p = {nowcast.M1_AR+BLP.cw_p}; {nowcast_n_forecasts} forecasts"),
    ("Study 3 · verifier", "ref",
     "{verifier.passed} of {verifier.sections} drafted sections pass; every injected error is blocked",
     "{verifier.sentences} sentences, citation coverage {verifier.coverage_min}; the negative control fails on numbers, citations and names"),
    ("Study 4 · power", "ref",
     "30 participants × 4 tasks give power {power_grid.4tasks_35pct.30} if the true time saving is 35%",
     "dry run: estimated reduction {dryrun.reduction}, 95% CI {dryrun.ci_lo} to {dryrun.ci_hi}"),
    ("BLP data", "null",
     "Results on the confidential BLP data pending",
     "every number here comes from the synthetic data generator and verifies the software, not the hypotheses"),
]

CREF = {"sec": ("Section", "Sections"), "app": ("Appendix", "Appendices"), "tab": ("Table", "Tables"),
        "fig": ("Figure", "Figures"), "eq": ("Equation", "Equations")}


def run(cmd: list[str], cwd: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True, text=True, **kw)


def lookup(numbers: dict, key: str) -> str:
    if key == "corr_raw" or key == "corr_adj":
        col = "BLP_RAI_raw" if key == "corr_raw" else "BLP_RAI"
        vals = [r[col] for r in numbers["corr_activity"].values()]
        return f"{sum(vals) / len(vals):.2f}"
    v = numbers
    for part in key.split("."):
        v = v[part]
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 0.01 else f"{v:.2f}" if abs(v) < 10 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def fill(s: str, numbers: dict) -> str:
    return re.sub(r"\{([\w.+]+)\}", lambda m: lookup(numbers, m.group(1)), s)


def labels(aux: str) -> dict[str, str]:
    """\\newlabel{sec:intro}{{1}{3}...} -> {"sec:intro": "1"} (cleveref's @cref entries skipped)."""
    return {k: v for k, v in re.findall(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}", aux)}


def cref_text(m: re.Match) -> str:
    """\\cref{sec:a,sec:b} -> Sections \\ref{sec:a} and \\ref{sec:b} (what cleveref prints)."""
    keys = [k.strip() for k in m.group(2).split(",")]
    name = CREF.get(keys[0].split(":")[0], ("", ""))[0 if len(keys) == 1 else 1]
    refs = [f"\\ref{{{k}}}" for k in keys]
    joined = refs[0] if len(refs) == 1 else ", ".join(refs[:-1]) + " and " + refs[-1]
    return f"{name}~{joined}".lstrip("~")


def tikz_svgs(body: str, preamble: str, work: Path, figdir: Path) -> tuple[str, list[str]]:
    """Compile each tikzpicture standalone and replace it by \\includegraphics of its SVG."""
    tikzset = re.search(r"\\tikzset\{.*?\n\}", preamble, re.DOTALL).group(0)
    libs = re.search(r"\\usetikzlibrary\{[^}]*\}", preamble).group(0)
    names: list[str] = []

    def one(m: re.Match) -> str:
        name = f"tikz_{len(names) + 1}"
        doc = (
            "\\documentclass[border=6pt]{standalone}\n\\usepackage[T1]{fontenc}\n\\usepackage{amsmath}\n"
            "\\IfFileExists{newtxtext.sty}{\\usepackage{newtxtext}\\usepackage{newtxmath}}{\\usepackage{mathptmx}}\n"
            f"\\usepackage{{xcolor}}\n\\usepackage{{tikz}}\n{libs}\n{tikzset}\n"
            f"\\begin{{document}}\n{m.group(0)}\n\\end{{document}}\n"
        )
        (work / f"{name}.tex").write_text(doc, encoding="utf-8")
        run(["tectonic", "-X", "compile", f"{name}.tex"], work)
        run(["pdftocairo", "-svg", f"{name}.pdf", str(figdir / f"{name}.svg")], work)
        names.append(f"{name}.svg")
        return f"\\includegraphics{{figures/{name}.svg}}"

    return re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", one, body, flags=re.DOTALL), names


def col_spec(spec: str) -> str:
    """tabularx/tabular column spec -> plain l/c/r letters pandoc understands."""
    spec = re.sub(r"[@>]\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", spec)
    spec = re.sub(r"[Lpmb]\{[^}]*\}|[YX]", "l", spec)
    return re.sub(r"[^lcr]", "", spec)


def simplify_tables(tex: str) -> str:
    tex = re.sub(r"\\begin\{tabularx\}\{[^}]*\}\{(.*?)\}\n", lambda m: f"\\begin{{tabular}}{{{col_spec(m.group(1))}}}\n", tex)
    tex = re.sub(r"\\begin\{tabular\}\{(@\{\}.*?@\{\})\}", lambda m: f"\\begin{{tabular}}{{{col_spec(m.group(1))}}}", tex)
    return tex.replace("\\end{tabularx}", "\\end{tabular}")


def pandoc(tex: str, cwd: Path, *extra: str) -> str:
    return run(["pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", *extra], cwd, input=tex).stdout


def build() -> Path:
    for tool in ("pandoc", "tectonic", "pdftocairo"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found; it is needed to build the web reader")
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    numbers = json.loads((PAPER / "numbers.json").read_text(encoding="utf-8"))
    figdir = OUT / "figures"
    if OUT.exists():
        shutil.rmtree(OUT)
    figdir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copytree(PAPER, work / "paper")
        run(["tectonic", "-X", "compile", "main.tex", "--keep-intermediates"], work / "paper")
        num = labels((work / "paper" / "main.aux").read_text(encoding="utf-8"))

        # Masthead pieces from the title block
        title = re.search(r"\{\\LARGE\\bfseries (.*?)\\par\}", tex, re.DOTALL).group(1).strip()
        subtitle = re.search(r"\{\\large\\bfseries (Text-Based.*?)\\par\}", tex, re.DOTALL).group(1)
        subtitle = re.sub(r"\\\\(\[[^]]*\])?\s*", " ", subtitle).strip()
        author = re.search(r"\\hypersetup\{.*?pdfauthor=\{([^}]*)\}", tex).group(1)
        date = re.search(r"\{\\small ([A-Z][a-z]+ \d{4})\}", tex).group(1)
        dot = re.compile(r"\s*\$\\cdot\$\s*")
        kw = dot.sub(", ", re.search(r"\\textbf\{Keywords\}\\quad (.*)", tex).group(1))
        jel = dot.sub(", ", re.search(r"\\textbf\{JEL classification\}\\quad (.*)", tex).group(1))
        abstract_tex = re.search(r"\\begin\{quote\}\s*\\small\s*(.*?)\\end\{quote\}", tex, re.DOTALL).group(1)
        abstract = pandoc(abstract_tex, work)

        # Body: Introduction .. end of document, rewritten into what pandoc reads well
        body = tex[tex.index("\\section{Introduction}"):tex.index("\\end{document}")]
        body = re.sub(r"\\(c|C)ref\{([^}]*)\}", cref_text, body)
        body, tikz = tikz_svgs(body, tex, work, figdir)
        body = re.sub(r"(\\includegraphics(?:\[[^]]*\])?\{figures/[\w-]+)\.pdf\}", r"\1.svg}", body)
        body = simplify_tables(body)
        body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "REFERENCES-HERE", body)
        body = body.replace("\\appendix", "APPENDIX-HERE")
        page = run(
            ["pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", "--standalone",
             f"--template={PAPER / 'reader.html'}", "--number-sections", "--toc", "--toc-depth=1",
             "--citeproc", "--bibliography=references.bib", "--metadata=link-citations:true",
             f"--metadata=pagetitle:{title}", f"--metadata=title:{title}", f"--metadata=subtitle:{subtitle}",
             f"--metadata=author:{author}", f"--metadata=date:{date}",
             f"--metadata=keywords:{kw}", f"--metadata=jel:{jel}",
             f"--metadata=description:Preprint: {title}. {subtitle}."],
            work / "paper", input="\\begin{document}\n" + body + "\n\\end{document}\n",
        ).stdout

    # Numbers as in the PDF: section headings (appendices are lettered), \ref links, captions
    def heading(m: re.Match) -> str:
        n = num.get(m.group(2))
        return m.group(0) if n is None else f'{m.group(1)}<span class="header-section-number">{n}</span>'

    page = re.sub(r'(<h1 data-number="[^"]*" id="([^"]+)"[^>]*>)<span class="header-section-number">[^<]*</span>', heading, page)
    page = re.sub(r'(<a href="#([^"]+)"[^>]*>)<span class="toc-section-number">[^<]*</span>',
                  lambda m: f'{m.group(1)}<span class="toc-section-number">{num.get(m.group(2), "")}</span>', page)
    page = re.sub(r'(<a href="#([^"]+)" data-reference-type="ref"[^>]*>)[^<]*</a>',
                  lambda m: f"{m.group(1)}{num.get(m.group(2), '??')}</a>", page)
    unresolved = re.findall(r'data-reference="([^"]+)">\?\?<', page)
    if unresolved:
        raise RuntimeError(f"unresolved references: {unresolved}")
    count = {"caption": 0, "figcaption": 0}

    def caption(m: re.Match) -> str:
        tag, attrs, para = m.group(1), m.group(2), m.group(3) or ""
        count[tag] += 1
        return f"<{tag}{attrs}>{para}<b>{'Table' if tag == 'caption' else 'Figure'} {count[tag]}.</b> "

    page = re.sub(r"<(caption|figcaption)(\b[^>]*)>(\s*<p>)?", caption, page)
    page = re.sub(r"<img\b", '<img loading="lazy"', page)
    page = re.sub(r'(<img[^>]*?)\s(?:style|width|height)="[^"]*"', r"\1", page)
    page = re.sub(r"<table\b[^>]*>\s*<caption>(.*?)</caption>(.*?)</table>",
                  r'<div class="tbl"><p class="tcap">\1</p><div class="tscroll"><table>\2</table></div></div>',
                  page, flags=re.DOTALL)
    page = re.sub(r"<colgroup>.*?</colgroup>\s*", "", page, flags=re.DOTALL)

    # References before the appendices, as in the PDF
    refs = re.search(r'<div id="refs".*?\n</div>\n', page, re.DOTALL)
    if refs:
        page = page.replace(refs.group(0), "")
        page = re.sub(r"<p>REFERENCES-HERE</p>",
                      lambda _: '<h1 class="unnumbered" id="references">References</h1>\n' + refs.group(0), page)
        page = re.sub(r'<li><a href="#refs"[^<]*</a></li>\s*', "", page)
    page = page.replace("<p>APPENDIX-HERE</p>", "")
    if "REFERENCES-HERE" in page or "APPENDIX-HERE" in page:
        raise RuntimeError("reference or appendix marker left in the page")

    items = "\n".join(
        f'<li class="r r-{style}"><span class="r-id">{html.escape(label)}</span>'
        f'<span class="r-text">{html.escape(fill(finding, numbers))}<small>{html.escape(fill(detail, numbers))}</small></span></li>'
        for label, style, finding, detail in RESULTS
    )
    bib = (
        "@misc{chongmankhong2026liaison,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{title}: {subtitle}}},\n"
        "  year   = {2026},\n"
        "  note   = {Preprint},\n"
        f"  url    = {{{PAPER_URL}}}\n"
        "}"
    )
    for marker, value in (
        ("<!--ABSTRACT-->", abstract),
        ("<!--ACTIONS-->", '<li><a href="paper.pdf">Download PDF</a></li>'
                           f'<li><a href="{REPO_URL}">Code and replication</a></li>'
                           f'<li><a href="{DATA_URL}">Data (Hugging Face)</a></li>'),
        ("<!--RESULTS-NOTE-->", "Headline numbers from paper/numbers.json. All of them come from the synthetic data "
                                "generator and validate the software, not the research hypotheses. Amber: results "
                                "on synthetic data; grey: pending."),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"),
        ("<!--COLOPHON-->", "<p>Rendered from <code>paper/main.tex</code> by <code>scripts/build_web.py</code> "
                            "(<code>make web</code>). Every number comes from the same source as the PDF, "
                            "which is authoritative. BLP data are confidential; this page uses synthetic data only.</p>"),
    ):
        if marker not in page:
            raise RuntimeError(f"reader.html lacks {marker}")
        page = page.replace(marker, value)
    if re.search(r"\{[\w.+]+\}|nan", items):
        raise RuntimeError("results strip has unresolved numbers")

    (OUT / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(PAPER / "main.pdf", OUT / "paper.pdf")
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page)) - set(tikz)):
        src = PAPER / "figures" / name.replace(".svg", ".pdf")
        run(["pdftocairo", "-svg", str(src), str(figdir / name)], ROOT)
    return OUT / "index.html"


if __name__ == "__main__":
    print(build().relative_to(ROOT))
