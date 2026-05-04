# Exporters

An `Exporter` takes an `AssembledReport` and writes one or more output
files.  Two implementations ship; `bus.export(..., backend="auto")`
picks the right one based on what's installed.

| Backend                 | Default? | Output formats                              | System dep |
|-------------------------|----------|---------------------------------------------|------------|
| `QuartoExporter`        | when Quarto on `$PATH` | HTML, PDF (Typst), DOCX, Reveal.js (citeproc citations, Bootswatch themes) | Quarto CLI |
| `WeasyPrintExporter`    | fallback | HTML, PDF (WeasyPrint), DOCX (via pypandoc), Reveal.js (CDN bundle) | none (pip-only) |

```python
paths = bus.export(
    formats=["html", "pdf", "revealjs"],
    output_dir="./out",
    title="Report",
    theme="cosmo",
    backend="auto",          # "auto" | "quarto" | "weasyprint"
    author="Marco Selvatici",
)
print(paths)
# {"html": Path("./out/report.html"),
#  "pdf":  Path("./out/report.pdf"),
#  "revealjs": Path("./out/report.revealjs.html")}
```

## QuartoExporter (primary)

### How it works

1. Renders the `AssembledReport` to a `.qmd` Markdown document via
   `quarto.qmd.render_report_to_qmd()`.
2. Writes a CSL-JSON bibliography (`refs.json`) when citations exist.
3. Writes a `_quarto.yml` project file declaring the requested formats,
   theme, and bibliography reference.
4. Pre-rasterises Plotly charts to PNG (Vega-Lite charts get rasterised
   by Quarto itself via the `{vegalite}` filter).
5. Shells out to `quarto render report.qmd --to <format>` once per
   format.  One subprocess invocation per format keeps stderr scoped
   and dodges Quarto's mixed-format quirks.

### Format → Quarto target

| `bus.export()` format | Quarto target | Output filename       |
|-----------------------|---------------|------------------------|
| `html`                | `html`        | `report.html`          |
| `pdf`                 | `typst`       | `report.pdf`           |
| `docx`                | `docx`        | `report.docx`          |
| `revealjs`            | `revealjs`    | `report.revealjs.html` |

PDF goes through Typst (no LaTeX install needed).  DOCX goes through
Pandoc citeproc which means citations resolve correctly there too.
Reveal.js is renamed to `report.revealjs.html` so it can coexist with
HTML in the same directory.

### Installing Quarto

```bash
# macOS
brew install quarto

# Linux (Debian/Ubuntu)
wget https://quarto.org/download/latest/quarto-linux-amd64.deb
sudo dpkg -i quarto-linux-amd64.deb

# Verify
quarto --version
```

### Themes

Quarto exposes the [Bootswatch](https://bootswatch.com/) theme catalog.
Pass any Bootswatch name as `theme=`:

```python
bus.export(["html"], "./out", title="…", theme="darkly")     # dark mode
bus.export(["html"], "./out", title="…", theme="lux")        # editorial
bus.export(["html"], "./out", title="…", theme="cosmo")      # default, clean
```

Known theme names: `cosmo`, `flatly`, `journal`, `litera`, `lumen`,
`lux`, `materia`, `minty`, `morph`, `pulse`, `quartz`, `sandstone`,
`simplex`, `sketchy`, `slate`, `solar`, `spacelab`, `superhero`,
`united`, `vapor`, `yeti`, `zephyr`, `cyborg`, `darkly`.

Unknown names pass through with a comment in `_quarto.yml` so a
misspelling is debuggable rather than silent.

### Errors

* `QuartoNotFoundError` — Quarto isn't on `$PATH`.  Install it, or use
  `backend="weasyprint"`.  When `backend="auto"`, this never raises;
  the resolver falls back automatically.
* `QuartoRenderError(returncode, stderr)` — `quarto render` exited
  non-zero.  The exception carries the captured stderr verbatim so you
  can paste it into a bug report.
* Subprocess timeout — 5 minutes per format.  Reports that legitimately
  take longer should be split.

## WeasyPrintExporter (fallback)

### How it works

* HTML: re-uses the existing `lazybridge/external_tools/report_builder/renderer.py`
  pipeline (Markdown → bleach → Jinja2 → one of 4 themes).  The
  fragment-side renderer assembles a Markdown document from the
  `AssembledReport` tree, then hands it to the existing renderer.
* PDF: WeasyPrint with `@page` rules for page numbers and
  `page-break-before: always` on top-level h2.
* DOCX: shells out to `pypandoc` (Pandoc binary required).  Citations
  render as a plain Sources list — no citeproc.
* Reveal.js: a static one-file HTML bundle pointing at the reveal.js
  CDN.  Honest second-best vs Quarto's real reveal output.

### Theme aliasing

The fallback uses the 4 legacy themes shipped with the extension
(`executive`, `financial`, `technical`, `research`).  Bootswatch theme
names map down to a sensible legacy theme:

| Bootswatch input | Legacy theme used |
|------------------|-------------------|
| `cosmo`, `flatly`, `lux` | `executive`       |
| `litera`, `morph`        | `research`        |
| `minty`                  | `financial`      |
| `darkly`, `slate`        | `technical`      |
| anything unknown         | `executive`      |

Pass a legacy name directly (`theme="research"`) to bypass aliasing.

### When you'd use it explicitly

* You can't install Quarto in your environment (locked-down container,
  edge worker, lambda).
* You're rendering many small reports per second and want to avoid the
  Quarto subprocess overhead.
* You're testing the pipeline end-to-end without a system dep.

```python
paths = bus.export(["html", "pdf"], "./out", title="…", backend="weasyprint")
```

### Errors

* `ImportError` — weasyprint or pypandoc missing.  Install
  `lazybridge[report-fallback]`.
* WeasyPrint runtime errors — the underlying library has known
  version-sensitivity (pydyf compatibility).  Pin
  `weasyprint>=62.0,<63.0` and matching pydyf if you hit
  `AttributeError` from inside WeasyPrint internals.

## Format-by-format reference

### HTML

* **Quarto**: Bootswatch theme, sticky TOC (`toc-location: left`),
  Vega-Lite charts render interactively via the `{vegalite}` filter.
  Plotly charts use the CDN script tag injected by the engine.
* **WeasyPrint**: legacy 4-theme catalog, Markdown rendered through
  `bleach` for safety.  Vega-Lite + Plotly embeds inlined as raw HTML.
  No interactive Quarto features.

### PDF

* **Quarto + Typst**: fast (~sub-second for a 20-page report), modern
  typography, no LaTeX.  Plotly figures pre-rasterised via kaleido when
  installed; without kaleido the figure becomes "(rendered in HTML
  only)".  Vega-Lite figures rasterise reliably via `vl-convert`.
* **WeasyPrint**: HTML + CSS Paged Media.  `@page` block adds page
  numbers `1 / N` in the footer; `h2` gets `page-break-before: always`
  except for the first.  No native chart rasterisation — charts render
  as the same HTML embed in a printable layout.

### DOCX

* **Quarto**: Pandoc citeproc citations, Word styles for headings and
  callouts, embedded charts (rasterised).  Probably the highest-quality
  DOCX output in the pipeline.
* **WeasyPrint** (via pypandoc): the same Markdown blob the HTML render
  uses, fed to `pypandoc.convert_text(md, "docx")`.  Acceptable for
  internal handoff; missing TOC fields and citeproc niceties.

### Reveal.js slides

* **Quarto**: uses `format: revealjs` — Quarto auto-paginates by `##`
  headings, themes via `theme: simple`/`black`/`white`/`league`/etc.
  Fragments get the `incremental` treatment when configured.
* **WeasyPrint** (one-file bundle): we emit a self-contained HTML
  document that loads reveal.js v5 from `cdn.jsdelivr.net`.  One slide
  per top-level section + child sections + a Sources slide.  No theme
  customisation; no fragments-in-fragments.

### When PPTX

PPTX isn't a first-class output.  The plan brief explicitly chose
Reveal.js over PPTX because Pandoc PPTX is the weakest of the four
Pandoc formats (template corruption with embedded data-URL PNGs,
"Could not find shape" errors with stock corporate templates) and
`python-pptx` is effectively abandoned.  If you really need PPTX:

* Render Reveal.js, then export to PPTX from PowerPoint or Keynote.
* Ask Quarto to render `pptx` directly (best-effort; expect manual
  cleanup for non-trivial reports).

## Picking a backend

| Need                                            | Backend     |
|-------------------------------------------------|-------------|
| Highest output quality across all formats       | Quarto      |
| `pip install` only, no system deps              | WeasyPrint  |
| Real citations + bibliography                   | Quarto      |
| Fast iteration during development               | WeasyPrint  |
| Reveal.js with proper themes                    | Quarto      |
| Render in a locked-down container               | WeasyPrint  |
| Auto-fallback (best when available, works always) | `backend="auto"` |

## Custom exporters

`Exporter` is an ABC with a single method:

```python
from lazybridge.external_tools.report_builder.exporters import Exporter
from lazybridge.external_tools.report_builder.assemblers import AssembledReport
from pathlib import Path

class MyExporter(Exporter):
    def export(
        self,
        report: AssembledReport,
        output_dir: Path,
        *,
        formats: list[str],
        theme: str = "cosmo",
    ) -> dict[str, Path]:
        ...
```

You can call it directly (skip `bus.export()`) or pass it positionally
once `bus.export(..., exporter=MyExporter())` is added — for now,
custom exporters are invoked manually:

```python
report = bus.assemble(title="Custom")
paths = MyExporter().export(report, Path("./out"), formats=["html"])
```

## See also

- [FragmentBus](fragment-bus.md)
- [Assemblers](report-assemblers.md)
- [Citations](report-citations.md)
- [Quarto docs](https://quarto.org/docs/)
- [WeasyPrint docs](https://doc.courtbouillon.org/weasyprint/)
